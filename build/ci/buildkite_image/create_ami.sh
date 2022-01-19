#!/bin/bash
set -eux

# Get the base AMI
BASE_AMI=$(python3 ./get_base_ami.py --aws-region ${AWS_REGION} --buildkite-stack-version ${BUILDKITE_STACK_VERSION})

# Get the instances public IP
set +x
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
PUBLIC_IP=`curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/public-ipv4`
LOCAL_IP=`curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/local-ipv4`
set -x

echo "Public IP: ${PUBLIC_IP}. Local IP: ${LOCAL_IP}"

docker run \
		-e AWS_ACCESS_KEY_ID \
		-e AWS_SECRET_ACCESS_KEY \
		-e PACKER_LOG \
		-v "${PWD}:/src" \
		--rm \
		-w /src \
		hashicorp/packer build -timestamp-ui -var "region=${AWS_REGION}" -var "source_ami=${BASE_AMI}" -var "public_ip=${PUBLIC_IP}" gpu_ami.json
