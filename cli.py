# -*- coding: utf-8 -*-
"""
# Created by Ruben Dörfel at 13.02.2021

Feature: #Test cli
"""
import sys
import argparse
parser = argparse.ArgumentParser(description='Add some integers.')

parser.add_argument('--length', action='store', type=int, default='5',
                    help='Length of data to take into account (0,5].')

parser.add_argument('--subjects', action='store', type=int, default='35',
                    help='Number of subjects to use [1,35].')

args = parser.parse_args()
N_sec = args.length
Ns = args.subjects

print("file_" + str(N_sec) + "_" + str(Ns))
