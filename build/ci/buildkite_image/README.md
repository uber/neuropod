Builds a buildkite image that has GPU drivers and the NVIDIA container toolkit installed

Make sure you have docker installed. You also need the `requests` and `cfn_flip` python packages installed

Instructions to run

```
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-2 BUILDKITE_STACK_VERSION=5.3.0 PACKER_LOG=1 ./create_ami.sh
```
