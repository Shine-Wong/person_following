x=dir('/home/wangshuai/test/useful_frames/*.jpg');   
for i=1:length(x)    
    x1=x(i).name;  
    x2=i;     
    a = sprintf('%d',i-1)      
    x3=num2str(a);
    x4=char('.jpg');  
    x5=strcat(x3,x4); 
    copyfile(['/home/wangshuai/test/useful_frames/' x1],['/home/wangshuai/test/useful_frames_with_index_sorted/' x5]);  
end  