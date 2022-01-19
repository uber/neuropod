import argparse
import requests

from cfn_tools import load_yaml


parser = argparse.ArgumentParser(description='Get the base AMI to use for a specific region and version of the buildkite elastic stack')
parser.add_argument('--aws-region', help='The AWS region (e.g. us-east-1)', required=True)
parser.add_argument('--buildkite-stack-version', help='The buildkite stack version (e.g. 5.3.0)', required=True)
args = parser.parse_args()

# Get the base AMI
r = requests.get(f"https://s3.amazonaws.com/buildkite-aws-stack/v{args.buildkite_stack_version}/aws-stack.yml")

if r.status_code != 200:
    raise ValueError("Failed to fetch buildkite stack config")

# Parse it and get the base AMI
config = load_yaml(r.text)
base_ami = config["Mappings"]["AWSRegion2AMI"][args.aws_region]["linuxamd64"]

print(base_ami)
