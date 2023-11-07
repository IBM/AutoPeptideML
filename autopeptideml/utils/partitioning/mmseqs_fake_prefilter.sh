#!/bin/bash

# copied from https://mmseqs.com/latest/userguide.pdf
# added LDB arg = local db path to make links work when using relative 
# paths. assumes pref is created in the same dir as db.
# e.g. when aligning temp/db to temp/db and creating temp/pref, link to db and not temp/db

fake_pref() {
    QDB="$1"
    TDB="$2"
    RES="$3"
    LDB="$4"
    # create link to data file which contains a list of all targets that should be aligned
    if [ -f $RES ]; then
        rm $RES
    fi
    ln -s "${LDB}.index" "${RES}"
    # create new index repeatedly pointing to same entry
    INDEX_SIZE="$(echo $(wc -c < "${TDB}.index"))"
    awk -v size=$INDEX_SIZE '{ print $1"\t0\t"size; }' "${QDB}.index" > "${RES}.index"
    # create dbtype (7)
    awk 'BEGIN { printf("%c%c%c%c",7,0,0,0); exit; }' > "${RES}.dbtype"
}

fake_pref "$@"