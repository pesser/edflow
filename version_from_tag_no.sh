#!/bin/sh

set -xe
TAG=$(git describe --abbrev=0 --tags)
echo $TAG

sed -i "s/dev/${TAG}/g" setup.py

cat setup.py
