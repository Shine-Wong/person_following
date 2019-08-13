after_path = '/home/wangshuai/PycharmProjects/followingRobot_code_V2_new/images_prepare_process/matlab_frames_with_new_background/';

fileid = fopen('conf.py','r')
low=0
up=0
frames_num=0
tline = fgetl(fileid)
while ischar(tline)
    tline=fgetl(fileid)
    if length(tline) < 10 && tline(1) == 'l' && tline(2) == 'o'
                low = str2num(tline(5:7))
    end
    if length(tline) < 10 && tline(1) == 'u'&& tline(2) == 'p'
                up = str2num(tline(4:6))
    end  
    if tline(1) == 'f'&& tline(2) == 'r'
                equation_index = strfind(tline,'=')     
                frames_num = str2num(tline(equation_index+1:length(tline)))
    end
    if tline(1) == 'b'&& tline(2) == 'a'
                equation_index = strfind(tline,'=')     
                backgrounds_num = str2num(tline(equation_index+1:length(tline)))
    end
end

    %for n=0:211
    for n=0:backgrounds_num-1
        for m=0:frames_num-1
            background_path_str=strcat('/home/wangshuai/PycharmProjects/followingRobot_code_V2_new/images_prepare_process/useful_background_with_index_sorted/',num2str(n),'.jpg');
            current_background = imread(background_path_str);
            frame_path_str=strcat('/home/wangshuai/PycharmProjects/followingRobot_code_V2_new/images_prepare_process/useful_frames_with_index_sorted/',num2str(m),'.jpg');
            current_frame = imread(frame_path_str);
            B = current_background;
            A = current_frame;
            [output_img,save_path]=put_in_new_background(A, B, m, n, after_path);
            save_path;
            imwrite(output_img,save_path,'jpg');
        end
    end
