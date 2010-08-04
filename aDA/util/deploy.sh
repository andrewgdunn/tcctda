#!/bin/bash
echo "=============================================="
echo "Cleaning up former results..."
rm $DXC_HOME/Scenarios/Results/*
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Clean up former Diagnostic Algorithm..."
rm $DXC_HOME/Algs/DiagnosticAlgorithm/*
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Move new Diagnostic Algorithm into framework..."
cp bin/* $DXC_HOME/Algs/DiagnosticAlgorithm/
echo "done."
echo "=============================================="


echo "\n=============================================="
echo "Job Complete"
echo "==============================================
