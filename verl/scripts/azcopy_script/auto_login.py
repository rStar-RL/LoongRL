#!/usr/bin/env python3
# ------------------------------------------------------------------
#  Author : Siyuan  ·  sywang0227@gmail.com
#  Note   : Simple helper script to keep Azure GPU sessions alive.
#           Runs `az login --use-device-code`, mails the device code,
#           and retries automatically on failure.  Keep it quiet. 🙂
# ------------------------------------------------------------------

import os, smtplib, subprocess, time
from datetime import datetime
from email.message import EmailMessage
from apscheduler.schedulers.blocking import BlockingScheduler

# ─── User configuration ───
COMMAND   = ["bash", "-c", 'echo "" | az login --use-device-code']
RUN_EVERY = dict(hours=8)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = "sywang0227@gmail.com"
SMTP_PASS = os.getenv("JOB_SMTP_PASS")
FROM_ADDR = SMTP_USER
TO_ADDRS  = ["wsy0227@sjtu.edu.cn", "v-gaozhang@microsoft.com"]
SUBJECT_TPL = "[KeepGPU Login] {ts}"
NODE_RANK = os.getenv("JOB_NAME", "Unknown job name") + " | " + os.getenv("NODE_RANK", "Unknown")
# NODE_RANK=os.getenv("NODE_RANK", "Unknown")  

DEVICE_URL_KEY = "https://microsoft.com/devicelogin"
QUIET_INTERVAL  = 5        # Seconds with no new output before flushing buffer
# ──────────────────────────


def send_email(subject: str, body: str):
    """Send a plain-text email via Gmail SMTP."""
    msg = EmailMessage()
    msg["From"] = FROM_ADDR
    msg["To"] = ", ".join(TO_ADDRS)
    msg["Subject"] = subject
    msg.set_content(body)
    # Ensure all recipients are included for SMTP
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as s:
        assert SMTP_PASS is not None, "Please set JOB_SMTP_PASS environment variable with your Gmail password."
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg, from_addr=FROM_ADDR, to_addrs=TO_ADDRS)


def attempt_login(subject: str) -> bool:
    """
    Start one az login attempt.
    Returns True on success (exit code 0), False if an ERROR line was detected
    and the attempt should be retried.
    """
    print(f"[{subject}] launching az login …")
    proc = subprocess.Popen(
        COMMAND,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=os.environ,
    )

    buffer: list[str] = []
    last_line_ts = time.time()

    while True:
        line = proc.stdout.readline()
        if line == "":                     # Sub-process ended
            break
        line = line.rstrip()
        print(line)
        now = time.time()

        # Send e-mail immediately when the device-code URL appears
        if DEVICE_URL_KEY in line:
            send_email(subject, f"{NODE_RANK}: Azure device login URL and code:\n\n{line}")
            print("[EMAIL] device-code sent")

        # Detect any ERROR line → terminate and request retry
        elif "ERROR" in line:
            send_email(
                subject,
                f"{NODE_RANK}:\n [KeepGPU] ERROR detected:\n\n{line}\n\n" + "\n".join(buffer),
            )
            print("[EMAIL] ERROR detected, will retry …")
            proc.terminate()
            proc.wait(timeout=10)
            return False

        else:
            buffer.append(line)

        # Flush buffer if no new line arrived within QUIET_INTERVAL
        if now - last_line_ts >= QUIET_INTERVAL:
            if buffer:
                send_email(subject, f"{NODE_RANK} [KeepGPU] Output since last block:\n\n" +
                           "\n".join(buffer))
                print(f"[EMAIL] flush {len(buffer)} lines")
                buffer.clear()
            last_line_ts = now
        else:
            last_line_ts = now
    proc.wait()
    # Process finished: flush remaining buffer and report exit code
    if buffer:
        send_email(subject, f"{NODE_RANK}:\n [KeepGPU] Remaining output:\n\n" + "\n".join(buffer))
        buffer.clear()

    summary = f"az login finished, exit code={proc.returncode}"
    send_email(subject, f"{NODE_RANK}:\n{summary}")
    print(summary)

    # On successful login, send next login schedule notice
    if proc.returncode == 0:
        from datetime import timedelta, timezone
        next_run_utc = datetime.now(timezone.utc) + timedelta(**RUN_EVERY)
        next_run_bj = next_run_utc + timedelta(hours=8)
        notice = (
            f"{NODE_RANK}:\n[KeepGPU] Login succeeded at {datetime.now().isoformat(timespec='seconds')}.\n\n"
            f"Next scheduled login attempt:\n"
            f"  • UTC : {next_run_utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  • BJT : {next_run_bj.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)\n"
        )
        send_email(subject, notice)

    return proc.returncode == 0


def run_and_notify_login():
    """Wrapper: keep retrying until one attempt succeeds."""
    ts = datetime.now().isoformat(timespec="seconds")
    subject = SUBJECT_TPL.format(ts=ts)

    while True:
        ok = attempt_login(subject)
        if ok:
            break                 # Successful login → stop retry loop
        time.sleep(5)             # Wait 5 s before starting a new attempt


# ─── Scheduler ───
if __name__ == "__main__":
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(run_and_notify_login, trigger="interval", **RUN_EVERY)
    print(f"Scheduler started – job every {RUN_EVERY}")
    try:
        run_and_notify_login()    # Run once immediately
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down…")