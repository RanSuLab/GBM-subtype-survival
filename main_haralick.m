imgPath =  'D:\feature_map\traditional_feature_extraction\tumor\';  % ͼ���ļ���·��
imgDir  = dir([imgPath '*.png']); % ��������jpg��ʽ�ļ�
haralick=[];
for i = 1:length(imgDir)          % �����ṹ��Ϳ���һһ����ͼƬ��
    
    feature = GLCM([imgPath imgDir(i).name])
    haralick=[haralick;feature]
    
end
csvwrite('haralick.csv',haralick);