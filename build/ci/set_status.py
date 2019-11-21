#
# Uber, Inc. (c) 2019
#

import argparse
import os
import subprocess
import requests

GH_STATUS_TOKEN = os.getenv("GH_STATUS_TOKEN")
GIT_COMMIT = os.getenv("TRAVIS_COMMIT", os.getenv("BUILDKITE_COMMIT"))

def set_status(sha, status):
    # https://api.github.com/repos/uber/neuropods/statuses/{sha}
    status_id = requests.post(
        'https://api.github.com/repos/uber/neuropods/statuses/{}'.format(sha),
        headers = {"Authorization": "token {}".format(GH_STATUS_TOKEN)},
        json = status
    ).json()["id"]
    print("Status ID: {}".format(status_id))

if __name__ == '__main__':
    # Make sure we have the information we need to continue
    if not GIT_COMMIT:
        raise ValueError("Could not get current commit")

    if not GH_STATUS_TOKEN:
        raise ValueError("Could not get GH token")

    parser = argparse.ArgumentParser()
    parser.add_argument('--context', help='The context of the github status', required=True)
    parser.add_argument('--description', help='The description of the github status', required=True)
    parser.add_argument('command', help='The command to run')
    parser.add_argument('args', help='The args to the command to run', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # The status to send to github
    status = {
        "state": "success",
        "description": args.description,
        "context": args.context,
    }

    # Try running the command
    command = [args.command] + args.args
    try:
        print("Running command:", command)
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        # The check failed
        status["state"] = "failure"

    print("Setting GH commit ({}) status: {}".format(GIT_COMMIT, status))
    set_status(GIT_COMMIT, status)
