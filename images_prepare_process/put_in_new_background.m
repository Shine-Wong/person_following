function [output_img,save_path]=put_in_new_background(A, B, m, n, after_path)
    %change the background of one single frame
    Ar = A(:,:,1);
    Ag = A(:,:,2);
    Ab = A(:,:,3);
    Br = B(:,:,1);
    Bg = B(:,:,2);
    Bb = B(:,:,3);
    %c = [192,192,192];
    %thresh = 115;
    c = [200,200,200];
    thresh = 125
    logmap = zeros([size(A,1),size(A,2)]);
    logmap = (Ar > (c(1)-thresh)).*(Ar < (c(1)+thresh)).*...
     (Ag > (c(2)-thresh)).*(Ag < (c(2)+thresh)).*...
     (Ab > (c(3)-thresh)).*(Ab < (c(3)+thresh));
    Ar(logmap == 1) = Br(logmap == 1);
    Ag(logmap == 1) = Bg(logmap == 1); 
    Ab(logmap == 1) = Bb(logmap == 1);
    A = cat(3 ,Ar,Ag,Ab);
    
    
    backg_dir_str=strcat(after_path,'background','_',num2str(n));
    if exist(backg_dir_str,'dir')==0
        mkdir(backg_dir_str);
    end
    save_path = strcat(backg_dir_str, '/' , num2str(m) , '_' , num2str(n) , '.jpg');
    output_img=A;
end
