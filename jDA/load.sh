#!/bin/bash

NEW_SCENARIO=$DXC_HOME/Doc/sample/Exp_"$1"_pb_ADAPT-Lite.scn

rm -f $DXC_HOME/Scenarios/Results/*
rm -f $DXC_HOME/Scenarios/ADAPT-Lite/*

cp $NEW_SCENARIO $DXC_HOME/Scenarios/ADAPT-Lite
