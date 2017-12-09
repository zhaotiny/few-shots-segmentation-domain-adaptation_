%imagefiles = dir('./labelsIdx_resize/*.png');  
clc;
clear;
close all;
imagefiles = dir('/home/selfdriving/zhaotiny/SegNet/cityscape/city_40_1/*.png');   %%% modify to the pred output
nfiles = length(imagefiles);    % Number of files found
imgSize = [480, 640];
load('./read_mapping/hitechMap.mat');
for ii=2:nfiles
   predFilenames = [imagefiles(ii).folder '/' imagefiles(ii).name];
   gtFilenames = strrep(predFilenames, '/city_40_1/', '/labelsIdx/'); % replace to get name of gt file
   oriImgFilename = strrep(predFilenames, '/city_40_1/', '/images/'); % replace to get name of pred file
 
   oriimage = imread(oriImgFilename);
   gtimage = imread(gtFilenames);
   predimage = imread(predFilenames);
   figure(1);
   subplot(2,2,1);
   imshow(oriimage);
   title('original image');
   subplot(2,2,2);
   imshow(gtimage, hitechMap);
   title('GT');
   subplot(2,2,3);
   imshow(predimage, hitechMap);
   title('prediction balance');

end