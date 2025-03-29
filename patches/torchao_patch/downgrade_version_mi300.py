import subprocess

def get_package_location(package_name):
    try:
        result = subprocess.run(['pip', 'show', package_name], capture_output=True, text=True, check=True)
        output = result.stdout
        for line in output.splitlines():
            if line.startswith('Location:'):
                location = line.split('Location:')[1].strip()
                return location

        return None
    except subprocess.CalledProcessError:
        print(f"Failed to find package: {package_name}")
        return None

if __name__ == '__main__':
    package_name = 'torchao'
    location = get_package_location(package_name)
    if location:
        print(f"The location of the package '{package_name}' is: {location}")
    else:
        print(f"Could not find location for package '{package_name}'.")

    old_string = 'TORCH_VERSION_AT_LEAST_2_6 = torch_version_at_least("2.6.0")'
    new_string = 'TORCH_VERSION_AT_LEAST_2_6 = torch_version_at_least("2.7.0")'
    print(f"replace line in torchao/utils.py\n    {old_string}\nto\n    {new_string}")

    file_path = location + '/torchao/utils.py'
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()

    file_contents = file_contents.replace(old_string, new_string)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(file_contents)

    print("Done.")
