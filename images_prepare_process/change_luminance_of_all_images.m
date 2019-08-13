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


after_path = '/home/wangshuai/PycharmProjects/followingRobot_code_V2/images_prepare_process/matlab_frames_with_new_background/';

    for n=0:backgrounds_num-1
        for m=0:frames_num-1
            original_path_str=strcat(after_path,'background','_',num2str(n),'/',num2str(m),'_',num2str(n),'.jpg');
            current_image = imread(original_path_str);
            [rows,cols,depth]=size(current_image);
            current_frame_hsv=rgb2hsv(current_image);
            current_channel_v=current_frame_hsv(:,:,3);
            random_v_change=low+(up-low).*rand(size(current_channel_v));
            new_channel_v=current_channel_v+random_v_change;
            new_img(:,:,1)=double(current_frame_hsv(:,:,1));
            new_img(:,:,2)=double(current_frame_hsv(:,:,2));
            new_img(:,:,3)=new_channel_v;
            new_img_bgr=hsv2rgb(new_img);

            backg_dir_str=strcat(after_path,'background','_',num2str(n+backgrounds_num));

            mkdir(backg_dir_str);
            save_path=strcat(backg_dir_str, '/' , num2str(m) , '_' , num2str(n+backgrounds_num) , '.jpg');
            imwrite(new_img_bgr,save_path,'jpg');
        end
    end
