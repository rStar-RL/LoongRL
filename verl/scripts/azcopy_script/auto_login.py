#!/usr/bin/env python3
import os
import smtplib
import subprocess
import textwrap
from datetime import datetime
from email.message import EmailMessage

from apscheduler.schedulers.blocking import BlockingScheduler

# ───────── CONFIG ─────────
COMMAND = ["bash", "-c", 'echo "" | az login --use-device-code']
RUN_EVERY = dict(hours=8)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = "sywang0227@gmail.com"
SMTP_PASS = os.getenv("JOB_SMTP_PASS")
FROM_ADDR = SMTP_USER
TO_ADDRS = ["wsy0227@sjtu.edu.cn"]
SUBJECT_TPL = "[KeepGPU Login] {ts}"

DEVICE_URL_KEY = "https://microsoft.com/devicelogin"
# ─────────────────────────

import os, smtplib, subprocess, time, re
from datetime import datetime
from email.message import EmailMessage

# DEVICE_URL_PAT  = re.compile(r"microsoft\.com/devicelogin", re.I)
DEVICE_URL_KEY = "https://microsoft.com/devicelogin"
QUIET_INTERVAL  = 5  # seconds: if no new output within this period, flush buffer

def send_email(subject: str, body: str):
    msg = EmailMessage()
    msg["From"] = FROM_ADDR
    msg["To"] = ", ".join(TO_ADDRS)
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.login(SMTP_USER, SMTP_PASS)
        smtp.send_message(msg)

def run_and_notify_login():
    ts = datetime.now().isoformat(timespec="seconds")
    subject = SUBJECT_TPL.format(ts=ts)
    print(f"[{ts}] Starting az login process...")

    process = subprocess.Popen(
        COMMAND,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=os.environ,
    )

    buffer: list[str] = []                     # Buffer to accumulate non-critical lines
    last_activity_ts: float = time.time()      # Last time we received output

    def flush_buffer(reason: str):
        """Send email with buffered lines if not empty, then clear the buffer."""
        nonlocal buffer
        if not buffer:
            return
        body = f"Triggered by: {reason}\n\n" + "\n".join(buffer)
        send_email(subject, body)
        print(f"[EMAIL] {reason}: sent {len(buffer)} lines")
        buffer = []

    # Read output line-by-line
    for line in iter(process.stdout.readline, ""):
        line = line.rstrip()
        print(line)
        found_key = False
        last_activity_ts = time.time()
        print(f"current line is {[line]}")
        current_time = time.time()
        if DEVICE_URL_KEY in line:
            found_key = True
            # If line includes device login URL → immediately send email
            body = "\n".join([line,])
            send_email(subject, f"Azure device login URL and code:\n\n{body}")
            print("[EMAIL] Device login code sent")
        else:
            buffer.append(line)
                # Append normal output to buffer
            last_buffer_time = current_time

            # Check quiet timeout
            while True:
                time.sleep(0.005)
                now = time.time()

                # If process has exited, break the loop
                if process.poll() is not None:
                    break

                # If time since last line > 3 seconds, flush buffer
                quiet_interval = 3
                if last_buffer_time and (now - last_buffer_time) >= quiet_interval:
                    if buffer:
                        body = "\n".join(buffer)
                        send_email(subject, f"[KeepGPU] Output since last block:\n\n{body}")
                        print(f"[EMAIL] Buffer sent: {len(buffer)} lines")
                        buffer.clear()
                    break  # Return to reading new lines


        # # If quiet for over 5 seconds → flush current output
        # if time.time() - last_activity_ts >= QUIET_INTERVAL:
        #     flush_buffer("5-second idle timeout")

    # Wait for the process to finish and flush any remaining output
    process.wait()
    flush_buffer("process finished")
    summary = f"az login finished with exit code {process.returncode}."
    send_email(subject, summary)
    print(f"[{ts}] az login complete — summary email sent.")


# ────── Scheduler ──────
if __name__ == "__main__":
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(run_and_notify_login, trigger="interval", **RUN_EVERY)
    print(f"Scheduler started – job every {RUN_EVERY}")
    try:
        run_and_notify_login()  # Run immediately once
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down…")