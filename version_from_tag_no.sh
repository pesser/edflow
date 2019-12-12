#!/bin/sh

set -xe
TAG=$(git describe --abbrev=0 --tags)

sed -i "s/dev/${TAG}/g" setup.py
