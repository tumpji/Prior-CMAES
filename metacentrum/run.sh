#!/bin/bash
# UTF-8
# =============================================================================
#         FILE: run.sh
#  DESCRIPTION:
#        USAGE:
#      OPTIONS:
# REQUIREMENTS:
#
#      LICENCE:
#
#         BUGS:
#        NOTES:
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2021 05.20.
# =============================================================================


. /storage/plzen1/home/tumpji/virtualni_env/bin/activate

cd /storage/plzen1/home/tumpji/ITAT2021/pokus2

python3 load_data.py
