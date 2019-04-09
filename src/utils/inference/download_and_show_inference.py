import time
from pathlib import Path

from src.estimator.launcher import providing_launcher
from src.utils import filenames
from src.utils.inference import secrets


def show_images_in_dir_via_eog(dir: Path, filename: str):
    from subprocess import call
    import os
    print("Showing images. Alternatively call:\n\n  cd {} && unzip -o ./{} && eog . --fullscreen\n ".format(dir,
                                                                                                            filename))
    os.chdir(str(dir))
    call(["unzip", "./{}".format(filename)])
    call(["eog", ".", "--fullscreen"])


def ssh_download_and_open(infer_dir):
    import paramiko
    with paramiko.SSHClient() as ssh_client:
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print("Connecting to remote host...")
        ssh_client.connect(hostname=secrets.REMOTE_HOST, username=secrets.USER, password=secrets.PASSWORD)

        filename = 'infer.zip'
        full_name = infer_dir / filename
        print("Zipping contests into {}".format(str(full_name)))
        (a, b, c) = ssh_client.exec_command("cd {}; zip -r {} ./".format(str(infer_dir), str(full_name)))
        print(b.readlines())
        print(c.readlines())

        with ssh_client.open_sftp() as ftp_client:
            localdir = Path("/tmp") / str(infer_dir.parts[-2]) / (
                        str(infer_dir.parts[-1]) + '_' + str(time.strftime('d%y%m%dt%H%M%S')))
            localdir.mkdir(parents=True, exist_ok=True)

            print("Downloading {} into {}".format(full_name, str(localdir / filename)))
            ftp_client.get(str(infer_dir / filename), str(localdir / filename))

    print("Connections closed.")
    show_images_in_dir_via_eog(localdir, filename)


if __name__ == '__main__':
    run_data = providing_launcher.provide_single_run_data()
    inference_dir = filenames.get_infer_dir(run_data)
    inference_dir = Path(str(inference_dir).replace('antek', 'ant'))
    ssh_download_and_open(inference_dir)
