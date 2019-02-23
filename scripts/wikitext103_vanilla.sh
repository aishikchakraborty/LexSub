#!/bin/bash

set -ex
export vanilla=True
export data="wikitext103"
. scripts/run_once.sh
