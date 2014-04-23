function a = imRead()

% A function that reads a set of images and converts it to feature vector
myFolder = '/home/manish/Contest/TagMe!-Data/Train/TestImages';
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, '*.jpg');
jpegFiles = dir(filePattern);
for k = 1:length(jpegFiles)
  baseFileName = jpegFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  imageArray = imread(fullFileName);
  a(:,:,k)=imageArray(:,:);
  imshow(imageArray);  % Display image.
  drawnow; % Force display to update immediately.
end

end

