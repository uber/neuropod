#
# Uber, Inc. (c) 2019
#

import os
import subprocess
import requests

NOW_DEPLOY_TOKEN = os.getenv("NOW_DEPLOY_TOKEN")
GH_STATUS_TOKEN = os.getenv("GH_STATUS_TOKEN")
GIT_COMMIT = os.getenv("TRAVIS_COMMIT", os.getenv("BUILDKITE_COMMIT"))
GIT_BRANCH = os.getenv("TRAVIS_BRANCH", os.getenv("BUILDKITE_BRANCH"))
IS_PR = (os.getenv("BUILDKITE_PULL_REQUEST", "false") != "false") or (os.getenv("TRAVIS_EVENT_TYPE") == "pull_request")


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

    if not NOW_DEPLOY_TOKEN:
        raise ValueError("Could not get now token")

    # Check if this is a prod release
    IS_PROD_BUILD = not IS_PR and GIT_BRANCH == "master"
    print("Deploying docs... Is prod build: {}".format(IS_PROD_BUILD))

    # The status to send to github
    status = {
        "state": "success",
        "description": "Docs published",
        "context": "docs/prod-publish" if IS_PROD_BUILD else "docs/staging-publish",
    }

    deploy_cmd = ["now", "-S", "neuropods", "-t", NOW_DEPLOY_TOKEN, "build/docs/"]
    if IS_PROD_BUILD:
        deploy_cmd += ["--prod"]

    # Try deploying the site
    try:
        status["target_url"] = subprocess.check_output(deploy_cmd).strip()
    except subprocess.CalledProcessError as e:
        # Error deploying
        status["state"] = "error"

    print("Setting GH commit ({}) status: {}".format(GIT_COMMIT, status))
    set_status(GIT_COMMIT, status)
