#!/bin/bash

syn=true ./scripts/run_once.sh
hyp=true ./scripts/run_once.sh
mer=true ./scripts/run_once.sh
syn=true hyp=true ./scripts/run_once.sh
syn=true mer=true ./scripts/run_once.sh
hyp=true mer=true ./scripts/run_once.sh
syn=true hyp=true mer=true ./scripts/run_once.sh
