close all
clear all
clc


load('C:\Users\Downloads\HKU-IS\HKU-IS\valImgSet.mat');

[nrows,ncols]= size(valImgSet);
fid=fopen('C:\Users\Downloads\HKU-IS\HKU-IS\valImgSet.txt','w'); 
for row=1:nrows
    fprintf(fid, '%s\n', valImgSet{row,:});
end
fclose(fid);
