"""
Created on Dec. 20, 2021
@author: Heng-Sheng (Hanson) Chang
"""

class POVRAYFrame:
    def __init__(self, included_files):
        self.included_files = included_files

    def write_included_files_to(self, file):
        for included_file in self.included_files:
            file.writelines('#include \"' + included_file + '\"\n')
