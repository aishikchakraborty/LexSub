#!/bin/bash

set -eux
export syn=true
export hyp=true
export mer=true
export data="wikitext2"
. scripts/run_once.sh
