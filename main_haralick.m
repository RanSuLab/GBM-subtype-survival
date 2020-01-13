imgPath =  'D:\feature_map\traditional_feature_extraction\tumor\';  % 图像文件夹路径
imgDir  = dir([imgPath '*.png']); % 遍历所有jpg格式文件
haralick=[];
for i = 1:length(imgDir)          % 遍历结构体就可以一一处理图片了
    
    feature = GLCM([imgPath imgDir(i).name])
    haralick=[haralick;feature]
    
end
csvwrite('haralick.csv',haralick);