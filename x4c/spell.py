import re

class Spell:
    ''' The Spell System

    A "spell" is a string that summarizes a series of data processing steps.
    A basic sentence should be in the form: "vn:ann_method:sa_method", where

    - `vn`: a variable name
    - `ann_method`: annualization method
    - `sa_method`: spatial average method

    One may also add more operations after the `vn` part:
    
    - "|plev": to interpolate the data from the model z levels to the pressure levels
    - "|regrid": to regrid the data from the model grid to the regular lat/lon grid

    '''
    def __init__(self, sentence:str):
        self.sentence = sentence

        self.alias = None
        self.vn = None
        self.ann_method = None
        self.sa_method = None

        self.slicing = None
        self.regrid = None
        self.plev = None
        self.zavg = None

        self.parse_alias()
        self.parse_sentence()
        self.parse_slicing()
        self.parse_regrid()
        self.parse_plev()
        self.parse_zavg()

    def parse_alias(self):
        if '~' in self.sentence:
            self.alias = self.sentence.split('~')[0].strip()
            self.sentence = self.sentence.split('~')[-1].strip()

    def parse_sentence(self):
        ''' A sentence should be
        '''
        if ':' not in self.sentence:
            self.vn = self.sentence
        else:
            basic_elements = self.sentence.split(':')
            if len(basic_elements) == 2:
                self.vn, self.ann_method = basic_elements
            elif len(basic_elements) == 3:
                self.vn, self.ann_method, self.sa_method = basic_elements
                if len(self.ann_method) == 0: self.ann_method = None

        if '|' in self.vn:
            self.vn = self.vn.split('|')[0]
    
    def parse_slicing(self):
        if '.' in self.vn:
            self.slicing = self.vn.split('.', 1)[-1]

    def parse_regrid(self):
        if '|regrid' in self.sentence:
            match = re.search(r'regrid\([^)]+\)', self.sentence)
            if match:
                self.regrid = match.group(0)
            else:
                self.regrid = 'regrid()'
            
    def parse_plev(self):
        if '|plev' in self.sentence:
            match = re.search(r'plev\([^)]+\)', self.sentence)
            if match:
                self.plev = match.group(0)
            else:
                self.plev = 'plev'

    def parse_zavg(self):
        if '|zavg' in self.sentence:
            match = re.search(r'zavg\([^)]+\)', self.sentence)
            if match:
                self.zavg = match.group(0)
            else:
                self.zavg = 'zavg()'