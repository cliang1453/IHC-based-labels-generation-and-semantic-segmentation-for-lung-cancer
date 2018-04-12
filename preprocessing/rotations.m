
[imfilenames,imfilepath]=uigetfile('*.tif;*.tiff;*.png','Select the Image file:','multiselect','on');

if ~iscell(imfilenames)
    imfilenames = {imfilenames};
end


for INDEX=1:length(imfilenames)
    
    
    imfilename = imfilenames{INDEX};
    img = imread([imfilepath imfilename]);
    
    img = imresize(img,[256,256]);
    
    [~,name,~]=fileparts(imfilename);
    
    imwrite(img,[imfilepath imfilename]);
    

    imwrite(rot90(img,1),strrep([imfilepath imfilename],'.png','_r90.png'));
    imwrite(rot90(img,2),strrep([imfilepath imfilename],'.png','_r180.png'));
    imwrite(rot90(img,3),strrep([imfilepath imfilename],'.png','_r270.png'));
end