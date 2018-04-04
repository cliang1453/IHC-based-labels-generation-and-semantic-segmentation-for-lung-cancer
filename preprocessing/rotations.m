
[imfilenames,imfilepath]=uigetfile('*.tif;*.tiff;*.png','Select the Image file:','multiselect','on');

if ~iscell(imfilenames)
    imfilenames = {imfilenames};
end


for INDEX=1:length(imfilenames)
    
    
    imfilename = imfilenames{INDEX};
    img = imread([imfilepath imfilename]);
    
    img = imresize(img,[256,256]);
    
    [~,name,~]=fileparts(imfilename);
    
    imwrite(img,colormap,[imfilepath imfilename]);
    
    imwrite(fliplr(img),colormap,strrep([imfilepath imfilename],'.png','_f.png'));
    imwrite(rot90(img,1),colormap,strrep([imfilepath imfilename],'.png','_r90.png'));
    imwrite(rot90(img,2),colormap,strrep([imfilepath imfilename],'.png','_r180.png'));
    imwrite(rot90(img,3),colormap,strrep([imfilepath imfilename],'.png','_r270.png'));
    imwrite(fliplr(rot90(img,1)),colormap,strrep([imfilepath imfilename],'.png','_r90_f.png'));
    imwrite(fliplr(rot90(img,2)),colormap,strrep([imfilepath imfilename],'.png','_r180_f.png'));
    imwrite(fliplr(rot90(img,3)),colormap,strrep([imfilepath imfilename],'.png','_r270_f.png'));
    
end