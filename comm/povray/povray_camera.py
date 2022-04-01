"""
Created on Dec. 20, 2021
@author: Heng-Sheng (Hanson) Chang
"""

class POVRAYCamera:
    def __init__(self, position, look_at, angle, rotate=[0.0, 0.0, 0.0], translate=[0.0, 0.0, 0.0], sky=[0.0, 0.0, 1.0], sun_position=[-1500, -1000, 2000], **kwargs):
        self.camera_position = "<%f, %f, %f>" % (position[0], position[1], position[2])
        self.camera_look_at = "<%f, %f, %f>" % (look_at[0], look_at[1], look_at[2])
        self.camera_angle = "%f" % (angle)
        self.camera_sky = "<%f, %f, %f>" % (sky[0], sky[1], sky[2])
        self.camera_rotate = "<%f, %f, %f>" % (rotate[0], rotate[1], rotate[2])
        self.camera_translate = "<%f, %f, %f>" % (translate[0], translate[1], translate[2])
        self.sun_position = "<%f, %f, %f>" % (sun_position[0], sun_position[1], sun_position[2])
        
        self.starting_string = '//--------------------------------------------------\n'
        self.ending_sting ='\n'

        self.default_setting = self.starting_string
        self.default_setting += '#version 3.6; // 3.7;\n'
        self.default_setting += 'global_settings{ assumed_gamma 1.0 }\n'
        self.default_setting += '#default{ finish{ ambient 0.1 diffuse 0.9 }}\n'
        self.default_setting += self.ending_sting

        self.default_included_lib_files = self.starting_string
        self.default_included_lib_files += '#include \"colors.inc\"\n'
        self.default_included_lib_files += '#include \"textures.inc\"\n'
        self.default_included_lib_files += '#include \"glass.inc\"\n'
        self.default_included_lib_files += '#include \"metals.inc\"\n'
        self.default_included_lib_files += '#include \"golds.inc\"\n'
        self.default_included_lib_files += '#include \"stones.inc\"\n'
        self.default_included_lib_files += '#include \"woods.inc\"\n'
        self.default_included_lib_files += '#include \"shapes.inc\"\n'
        self.default_included_lib_files += '#include \"shapes2.inc\"\n'
        self.default_included_lib_files += '#include \"functions.inc\"\n'
        self.default_included_lib_files += '#include \"math.inc\"\n'
        self.default_included_lib_files += '#include \"transforms.inc\"\n'
        self.default_included_lib_files += self.ending_sting

        self.camera_setting = self.starting_string
        self.camera_setting += '#declare Camera_Position  = ' + self.camera_position + ';\n'
        self.camera_setting += '#declare Camera_Look_At   = ' + self.camera_look_at + ';\n'
        self.camera_setting += '#declare Camera_Sky       = ' + self.camera_sky + ';\n'
        self.camera_setting += '#declare Camera_Angle     = ' + self.camera_angle + ';\n'
        self.camera_setting += '#declare Camera_Rotate    = ' + self.camera_rotate + ';\n'
        self.camera_setting += '#declare Camera_Translate = ' + self.camera_translate + ';\n'
        self.camera_setting += 'camera{ location  Camera_Position\n'
        self.camera_setting += '        right     -x*image_width/image_height\n'
        self.camera_setting += '        sky       Camera_Sky\n'
        self.camera_setting += '        angle     Camera_Angle\n'
        self.camera_setting += '        look_at   Camera_Look_At\n'
        self.camera_setting += '        rotate    Camera_Rotate\n'
        self.camera_setting += '        translate Camera_Translate\n'
        self.camera_setting += '}\n\n'
        self.camera_setting += '#declare Sun_Position = ' + self.sun_position + ';\n'
        self.camera_setting += 'light_source{ Sun_Position color White}                      // sun light\n'
        self.camera_setting += 'light_source{ Camera_Position color rgb<0.9, 0.9, 1.0>*0.1}  // flash light\n'
        self.camera_setting += 'background { color White }\n'
        self.camera_setting += self.ending_sting

        if kwargs.get('floor', False):
            self.camera_setting += 'declare RasterScale = 2.0;\n'
            self.camera_setting += 'declare RasterHalfLine  = 0.05;\n'
            self.camera_setting += 'declare RasterHalfLineZ = 0.05;\n'
            self.camera_setting += 'declare Raster_Color = <1,1,1>*0.2;\n'
            self.camera_setting += '#macro Raster(RScale, HLine)\n'
            self.camera_setting += 'pigment{\n'
            self.camera_setting += '\tgradient x scale RScale\n'
            self.camera_setting += '\tcolor_map{\n'
            self.camera_setting += '\t\t[0.000   color rgb Raster_Color]\n'
            self.camera_setting += '\t\t[0+HLine color rgb Raster_Color]\n'
            self.camera_setting += '\t\t[0+HLine color rgbt<1,1,1,1>]\n'
            self.camera_setting += '\t\t[1-HLine color rgbt<1,1,1,1>]\n'
            self.camera_setting += '\t\t[1-HLine color rgb Raster_Color]\n'
            self.camera_setting += '\t\t[1.000   color rgb Raster_Color]\n'
            self.camera_setting += '\t}// end of color_map\n'
            self.camera_setting += '}// end of pigment\n'
            self.camera_setting += '#end// of Raster(RScale, HLine)-macro\n'
            self.camera_setting += 'plane { <0,0,-1>, 1.0 \n'
            self.camera_setting += '\ttexture { pigment{color White}\n'
            self.camera_setting += '\t\tfinish {ambient 0.45 diffuse 0.85}}\n'
            self.camera_setting += '\ttexture { Raster(RasterScale,RasterHalfLine ) rotate<0,0,0> }\n'
            self.camera_setting += '\ttexture { Raster(RasterScale,RasterHalfLineZ) rotate<0,0,90>}\n'
            self.camera_setting += '\trotate<0,0,30>\n'
            self.camera_setting += '}\n'
    
    def write_to(self, file,):
        file.writelines(self.default_setting)
        file.writelines(self.default_included_lib_files)
        file.writelines(self.camera_setting)