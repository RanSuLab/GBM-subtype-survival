function [feature] = GLCM(readPath)
%HARALICK 此处显示有关此函数的摘要
%   此处显示详细说明
feature=[];
prot = imread(readPath);
Gray=rgb2gray(prot);

offsets = [0 1; -1 1; -1 0; -1 -1];
GLCM2 = graycomatrix(Gray,'NumLevels',16,'Offset',offsets);
stats = Get_feature(GLCM2);
feature = Struct_to_matrix(stats);
feature=reshape(feature,1,52);
