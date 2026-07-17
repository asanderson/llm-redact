# Terraform for a small service. Dense with identifier-shaped values that
# must NOT fire: resource ids, image digests, module hashes, ARNs (Bedrock
# non-target), and version pins. Nothing here is a secret.

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "5.63.1"
    }
  }
  backend "s3" {
    bucket = "acme-tfstate-prod"
    key    = "svc/redactor/terraform.tfstate"
    region = "us-east-1"
  }
}

# Image pinned by digest (sha256, 64 hex) — not a secret, must not fire.
locals {
  image  = "ghcr.io/acme/redactor@sha256:3f1b0c9e2d4a6b8c0e2f4a6b8d0c2e4f6a8b0d2c4e6f8a0b2d4c6e8f0a2b4c6d"
  module = "git::https://example.com/mods//net?ref=1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"
}

# A model ARN (percent-decoded on the Bedrock path) is provider config, not
# a value we redact.
variable "model_arn" {
  default = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude"
}

resource "aws_instance" "app" {
  ami           = "ami-0abcdef1234567890"
  instance_type = "t3.small"
  subnet_id     = "subnet-0f1e2d3c4b5a69788"
  tags = {
    Name    = "redactor"
    Commit  = "9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d"
    Managed = "terraform"
  }
}

output "private_ip" {
  # A documentation RFC1918 address in a comment: 10.x is default-allowlisted.
  value = aws_instance.app.private_ip
}
