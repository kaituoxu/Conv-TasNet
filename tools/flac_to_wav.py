import ffmpeg  # this is actually ffmpeg-python
import os

""" This can only be run in Linux"""


input_path = "../egs/wsj0/flac_data"
output_path = "../egs/wsj0/wsj0-hin/tr"
print('Starting flac to wav')
i = 0
for dir_name, sub_dir_list, file_name in os.walk(input_path):
    if file_name is not None:
        for file_flac in file_name:
            input_file = os.path.join(dir_name, file_flac)
            file_without_suffix, suffix = os.path.splitext(file_flac)
            if suffix == '.flac':
                i += 1
                output_file = os.path.join(output_path, file_without_suffix + ".wav")
                stream = ffmpeg.input(input_file)
                stream = ffmpeg.output(stream, output_file)
                ffmpeg.run(stream)
print('Done flac to wav, number of files:' + str(i))
#

