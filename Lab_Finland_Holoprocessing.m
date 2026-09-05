%% SETTINGS & CONFIGURATION
parentfolder = 'D:\Celeste_InternProject\Holoprocessing'; % Adjust path as needed
resolution = 2.14; % Image resolution microns per pixel
lambda = 0.532; % Laser wavelength in microns (Green)
invertsplats = 1; % 1 = White particles, Black background
save_diagnostics = 1; % Set to 1 to save Big and Binary images for EVERY processed frame
use_artifact_mask = 1; % 1 = Apply mask to remove ripples/lines
mask_file = fullfile(parentfolder, 'Raw', 'selected_mask.mat'); % Points to the mask you drew
%% Directory Setup & Flags
tic
rawfolder = fullfile(parentfolder, '\Raw\');
splatfolder = fullfile(parentfolder, '\Splat\');
subtractedfolder = fullfile(parentfolder, '\Subtracted\');
supplementfolder = fullfile(parentfolder, '\Supplementary\');
cropfolder = fullfile(parentfolder, '\Crop\');
diagnosticfolder = fullfile(parentfolder, '\Diagnostics\');
% Crop Flags
crop_flag = 1; % 1 to store cropped regions of interest
paddingNum = 16; % Padding for cropped images in pixels
rotate = 1; % 1 to rotate crops based on orientation
if rotate == 1
paddingNum = paddingNum * 10;
end
save_recon_planes = 0; % 1 to store recon planes
imx = 255; % Scale image intensity
% Create Directories
if ~exist(subtractedfolder, 'dir'), mkdir(subtractedfolder); end
if ~exist(splatfolder, 'dir'), mkdir(splatfolder); end
if ~exist(supplementfolder, 'dir'), mkdir(supplementfolder); end
if ~exist(cropfolder, 'dir'), mkdir(cropfolder); end
if save_diagnostics == 1 && ~exist(diagnosticfolder, 'dir') % <--- Add this
mkdir(diagnosticfolder);
end
if ~exist(fullfile(parentfolder, '\ReconstructedPlanes\'), 'dir') && save_recon_planes
mkdir(fullfile(parentfolder, '\ReconstructedPlanes\'));
end
%% Raw File Loading
rawfiles = filelist(rawfolder, "*.tif", 'fullpath', 1);
if isempty(rawfiles)
error('No TIFF files found in the Raw folder.');
end
disp(['Found ', num2str(length(rawfiles)), ' images to process.']);
%% Batch Background Subtraction
batchSize = 100; % You can adjust this. Smaller = more sensitive to changing backgrounds
batchID_lookup = ceil((1:length(rawfiles))' / batchSize);
numBatches = max(batchID_lookup);
% 1. Generate the localized background models
for b = 1:numBatches
bgFile = fullfile(supplementfolder, "background_batch" + b + ".mat");
batchIndices = find(batchID_lookup == b);
batchFiles = rawfiles(batchIndices);
if ~isfile(bgFile)
disp("Computing background for batch " + b);
numBG = min(length(batchFiles), 10); % Takes 10 sample images per batch
sampleIdx = round(linspace(1, length(batchFiles), numBG));
I_ave = double(imread(batchFiles(sampleIdx(1))));
I_min = I_ave; I_max = I_ave;
for i = sampleIdx(2:end)
I = double(imread(batchFiles(i)));
I_ave = I_ave + I;
I_min = min(I, I_min); I_max = max(I, I_max);
end
I_ave = I_ave / numBG;
I_min = I_min - I_ave; I_max = I_max - I_ave;
lowerBound = min(I_min(:)); upperBound = max(I_max(:));
save(bgFile, 'I_ave', 'lowerBound', 'upperBound');
end
end
% 2. Apply the localized models to the images
subtractedfiles = filelist(subtractedfolder, "*.tif", 'fullpath', 1);
if length(subtractedfiles) < length(rawfiles)
disp("Starting batch background subtraction...");
for i = 1:length(rawfiles)
batchID = batchID_lookup(i);
bgFile = fullfile(supplementfolder, "background_batch" + batchID + ".mat");
load(bgFile, 'I_ave', 'lowerBound', 'upperBound');
I = double(imread(rawfiles(i)));
I = I - I_ave;
I = (I - lowerBound) / (upperBound - lowerBound);
[~, rawimagename, ~] = fileparts(rawfiles(i));
putsubtractedimage = fullfile(subtractedfolder, rawimagename + "_subtracted.tif");
I = I(:,:,1);
I = uint8(I * 255);
imwrite(I, putsubtractedimage);
end
end
% Organize subtracted files numerically
subtractedfiles = filelist(subtractedfolder, "*.tif", 'fullpath', 1);
subtractedfiles = string(subtractedfiles);
imgNums = zeros(length(subtractedfiles),1);
for i = 1:length(subtractedfiles)
tokens = regexp(subtractedfiles(i), '\d+', 'match');
if ~isempty(tokens)
imgNums(i) = str2double(tokens{end});
else
imgNums(i) = Inf;
end
end
[~, idx] = sort(imgNums);
subtractedfiles = subtractedfiles(idx);
disp("All Images Background Subtracted!");
%% RECONSTRUCTION SETUP
refind = 1.33;
recon_planes = [0:200:40000]/refind;
if length(resolution) == 2
dy = resolution(1); dx = resolution(2);
elseif length(resolution) == 1
dy = resolution; dx = resolution;
else
error('resolution must be scalar or a 2 element vector');
end
I_test = imread(rawfiles(1));
Nx = size(I_test,2);
Ny = size(I_test,1);
padnum = 0;
I_test = padarray(I_test, [(Nx-Ny)/2+padnum padnum], 'symmetric');
Ny1 = Ny; Nx1 = Nx;
Ny = Nx + 2*padnum;
Nx = Nx + 2*padnum;
% Cropping filtering parameters8 task
filter_size = 15;
Sigma = 3;
thresholdValue1 = 50;
minParticleSize = 300;
preview_frame = 17;
halo_blur_sigma = 90;
quick_run_limit = 9999; % Set to exactly how many images you want to test or any number bigger than your set for full runs
num_images = min([length(rawfiles), length(subtractedfiles), quick_run_limit]);
%% MAIN RECONSTRUCTION LOOP
for f = 1:num_images
% Use to skip other images and save on preview time
%for f = preview_frame:preview_frame
I = double(imread(subtractedfiles{f}));
if use_artifact_mask == 1 && exist(mask_file, 'file')
maskData = load(mask_file);
art_mask = maskData.mask;
if size(I,1) == size(art_mask,1) && size(I,2) == size(art_mask,2)
% Instead of setting to 0 (which causes aggressive FFT ringing halos),
% we set the artifact zones to the mean background gray level.
I(art_mask) = mean(I(:));
else
warning('Mask dimensions do not match the image. Skipping masking for this frame.');
end
end
I = padarray(I, [(Nx1-Ny1)/2+padnum padnum], 'symmetric');
% High-pass filter
IMG = fftshift(fft2(I));
[Col, Row] = meshgrid(-floor(Nx/2):floor(Nx/2)-1, -floor(Ny/2):floor(Ny/2)-1);
Radius = sqrt(Col.^2 + Row.^2);
MaskRadius = 15;
ButterworthOrder = 3;
FilterMask = 1./(1+(MaskRadius./Radius).^(2*ButterworthOrder));
IMG(abs(IMG) > 0.9*max(max(abs(IMG)))) = 0;
IMG_Filtered = IMG .* FilterMask;
img_filtered = real(ifft2(ifftshift(IMG_Filtered)));
img_filtered = specklefilt(img_filtered, 'DegreeOfSmoothing', 0.1, 'NumIterations', 10);
img_filtered = (img_filtered - min(img_filtered(:))) / (max(img_filtered(:)) - min(img_filtered(:)));
img_filtered = imadjust(img_filtered, [max(0,(mean(img_filtered(:))-5*std(img_filtered(:)))) min(1,mean(img_filtered(:))+5*std(img_filtered(:)))], [0 1]);
Image = imx * img_filtered;
% Filename parsing
[~, fstr] = fileparts(rawfiles{f});
partsofname = string(strsplit(fstr, {'_','-'}));
imgnum = partsofname(1);
% GPU variables for Z-propagation
I_gpu = gpuArray(single(Image));
I_fft = fft2(I_gpu);
M1 = gpuArray.ones(Ny, Nx) * imx.^5;
S = gpuArray.zeros(Ny, Nx);
Tmax = gpuArray.zeros(Ny, Nx);
D2 = zeros(Ny, Nx); % Depth map needed for localized cropping later
Mx = [-1 0 1; -2 0 2; -1 0 1];
My = [-1 -2 -1; 0 0 0; 1 2 1];
x = gpuArray(((1:Nx)-Nx/2)/(Nx*dx));
y = gpuArray(((1:Ny)-Ny/2)/(Ny*dy));
[x, y] = meshgrid(x, y);
% Propagate through planes
for z = recon_planes
n = exp(-1i * lambda * pi * z * (x.^2 + y.^2));
n = single(ifftshift(n));
M = abs(ifft2(I_fft .* n));
T = sqrt(conv2(M, Mx, 'same').^2 + conv2(M, My, 'same').^2);
ind1 = M < M1;
M1(ind1) = M(ind1);
ind = T >= Tmax;
Tmax(ind) = T(ind);
S(ind) = M(ind);
D2(ind) = z; % Store optimal depth per pixel
end
% Strip padding to return to original dimensions
M1 = M1((Nx1-Ny1)/2+1+padnum:end-(Nx1-Ny1)/2-padnum, 1+padnum:end-padnum).^0.5;
S = S((Nx1-Ny1)/2+1+padnum:end-(Nx1-Ny1)/2-padnum, 1+padnum:end-padnum);
Tmax = Tmax((Nx1-Ny1)/2+1+padnum:end-(Nx1-Ny1)/2-padnum, 1+padnum:end-padnum);
D2 = D2((Nx1-Ny1)/2+1+padnum:end-(Nx1-Ny1)/2-padnum, 1+padnum:end-padnum);
Image = Image((Nx1-Ny1)/2+1+padnum:end-(Nx1-Ny1)/2-padnum, 1+padnum:end-padnum);
S = S .* M1;
% Halo Reduction
S_cpu = gather(S);
blurred_halo = imgaussfilt(S_cpu, halo_blur_sigma);
S = gpuArray(S_cpu ./ blurred_halo);
% Normalize Splats
Si = (S - min(S(:))) / (max(S(:)) - min(S(:)));
Si = imadjust(Si, [max(0,(mean(Si(:))-4*std(Si(:)))) min(1,mean(Si(:))+3*std(Si(:)))], [0 1], 0.9);
if invertsplats
Si = 0.7 * imx - Si .* imx;
end
Si = uint8(gather(Si));
% Save Full Splat Image
first_pos = find(subtractedfiles{f} == '\', 1, 'last') + 1;
last_pos = find(subtractedfiles{f} == '.', 1, 'last') - 1;
imgName = subtractedfiles{f}(first_pos:last_pos);
imgBase = erase(imgName, "_subtracted");
parts = strsplit(imgBase, "_image_");
prefix = parts{1};
imgnum_clean = strsplit(parts{2}, "_");
imgnum_clean = imgnum_clean{end};
imwrite(Si, fullfile(splatfolder, prefix + "_splat2_image_" + imgnum_clean + ".tif"), 'tif', 'compression', 'none');
%% Diagnostic & Preview Capture
if save_diagnostics == 1 || f == preview_frame
% 1. Generate the masks
filtered_preview = wiener2(Si, [filter_size filter_size]);
filtered_preview = gather(imgaussfilt(filtered_preview, Sigma, 'FilterSize', [3 3]));
Bin1_preview = filtered_preview > thresholdValue1;
thresh_val = adaptthresh(filtered_preview, 0.4, "NeighborhoodSize", 401, 'ForegroundPolarity', 'bright');
Bin2_preview = imbinarize(filtered_preview, thresh_val);
sample_BinaryImage = bwareaopen(Bin1_preview .* Bin2_preview, minParticleSize);
sample_bigImage = Si; % Start with the clean splat
% 2. Draw the red bounding boxes
cc_preview = bwconncomp(sample_BinaryImage);
stats_preview = regionprops(cc_preview, 'BoundingBox');
for obj_p = 1:length(stats_preview)
sample_bigImage = insertObjectAnnotation(sample_bigImage, "rectangle", stats_preview(obj_p).BoundingBox, obj_p, 'Color', 'red');
end
% 3. Save to hard drive if Diagnostics are toggled ON
if save_diagnostics == 1
bin_name = fullfile(diagnosticfolder, prefix + "_BinaryMask_image_" + imgnum_clean + ".tif");
big_name = fullfile(diagnosticfolder, prefix + "_BigSplat_image_" + imgnum_clean + ".tif");
% Multiply binary array by 255 so it saves properly as a black/white image
imwrite(uint8(sample_BinaryImage)*255, bin_name, 'tif', 'compression', 'none');
imwrite(sample_bigImage, big_name, 'tif', 'compression', 'none');
end
end
%% CROPPING SECTION
if crop_flag == 1
% 1. Create the sequentially numbered subfolder (00000, 00001, etc.)
current_crop_folder = sprintf('%05d', f-1);
targetCropDir = fullfile(cropfolder, current_crop_folder);
if ~exist(targetCropDir, 'dir')
mkdir(targetCropDir);
end
bigImage = Si;
filtered = wiener2(bigImage, [filter_size filter_size]);
filtered = gather(imgaussfilt(filtered, Sigma, 'FilterSize', [3 3]));
BinaryImage1 = filtered > thresholdValue1;
thresholdValue = adaptthresh(filtered, 0.2, "NeighborhoodSize", 401, 'ForegroundPolarity', 'bright');
BinaryImage2 = imbinarize(filtered, thresholdValue);
BinaryImage = BinaryImage1 .* BinaryImage2;
filteredBinaryImage = bwareaopen(BinaryImage, minParticleSize);
cc = bwconncomp(filteredBinaryImage);
% Restored properties necessary for accurate bounds and extraction
stats = regionprops(cc, 'BoundingBox', 'Orientation', 'PixelIdxList', 'PixelList', 'EquivDiameter');
coords = [];
for obj = 1:length(stats)
% 2. Restored BoundingBox logic
xMin = max(1, floor(stats(obj).BoundingBox(1) - paddingNum/2));
xMax = min(Nx1, ceil(stats(obj).BoundingBox(1) + stats(obj).BoundingBox(3) + paddingNum/2));
yMin = max(1, floor(stats(obj).BoundingBox(2) - paddingNum/2));
yMax = min(Ny1, ceil(stats(obj).BoundingBox(2) + stats(obj).BoundingBox(4) + paddingNum/2));
% Extract isolated object arrays
list = stats(obj).PixelIdxList;
D_sub = D2(list);
T_sub = Tmax(list);
% 3. Restored dynamic thresholding for precise focal depth
dum = 2:-0.05:1;
threshold = max(0.5*max(max((T_sub))), dum*mean(mean((T_sub))));
n_thresh = zeros(1, length(threshold));
for count = 1:length(threshold)
n_thresh(count) = nnz(T_sub > threshold(count));
end
para = max(80, 0.15 * nnz(T_sub));
if max(n_thresh) > para
threshold = threshold(find(n_thresh > para, 1));
else
continue; % skip small/noisy objects
end
depth = gather(mean(D_sub(T_sub > threshold)));
coords = [coords; obj, xMin, xMax, yMin, yMax, stats(obj).EquivDiameter, depth];
% 4. Restored finer-scale reconstruction localized strictly to the crop bounds
nx = xMax - xMin + 1;
ny = yMax - yMin + 1;
xc = gpuArray(((1:nx)-nx/2)/(nx*dx));
yc = gpuArray(((1:ny)-ny/2)/(ny*dy));
[xc, yc] = meshgrid(xc, yc);
I_crop = gpuArray(Image(yMin:yMax, xMin:xMax));
I_crop = fft2(I_crop);
n_crop = exp(-1i * lambda * pi * depth * (xc.^2 + yc.^2));
n_crop = single(ifftshift(n_crop));
M_crop = abs(ifft2(I_crop .* n_crop));
M_crop = (M_crop - min(M_crop(:))) / (max(M_crop(:)) - min(M_crop(:)));
M_crop = imadjust(M_crop, [0 min(1, mean(M_crop(:)) + 5*std(M_crop(:)))], [0 1]);
if invertsplats
cropImg = uint8(0.7 * imx - M_crop .* imx);
else
cropImg = uint8(M_crop .* imx);
end
if rotate == 1
theta = [stats(obj).Orientation];
transl = [0 0];
tform = rigidtform2d(theta, transl);
X = transformPointsForward(tform, stats(obj).PixelList - [xMin, yMin]);
centerOutput = affineOutputView(size(cropImg), tform, "BoundsStyle", "CenterOutput");
cropImg = imwarp(cropImg, tform, "OutputView", centerOutput, "FillValues", imx*mean(thresholdValue(:)));
% Adjust bounds after rotation
xMin_rot = max(1, floor(min(X(:,1)) - centerOutput.XWorldLimits(1) - paddingNum/20));
xMax_rot = min(centerOutput.ImageSize(2), ceil(max(X(:,1)) - centerOutput.XWorldLimits(1) + paddingNum/20));
yMin_rot = max(1, floor(min(X(:,2)) - centerOutput.YWorldLimits(1) - paddingNum/20));
yMax_rot = min(centerOutput.ImageSize(1), ceil(max(X(:,2)) - centerOutput.YWorldLimits(1) + paddingNum/20));
cropImg = cropImg(yMin_rot:yMax_rot, xMin_rot:xMax_rot);
end
bigImage = insertObjectAnnotation(bigImage, "rectangle", stats(obj).BoundingBox, obj, 'Color', 'red');
% 5. Write specific object crop to its dedicated subfolder
crop_filename = fullfile(targetCropDir, prefix + "_image_" + imgnum_clean + "_crop_" + sprintf('%04d', obj) + ".tif");
imwrite(cropImg, crop_filename, 'tif', 'compression', 'none');
end
% 6. Save total coordinates metrics inside the subfolder
if ~isempty(coords)
total_coords = size(coords, 1);
save(fullfile(targetCropDir, "Total_Coordinates.txt"), 'total_coords', '-ascii');
save(fullfile(targetCropDir, "Coordinates.txt"), 'coords', '-ascii');
end
% Update the preview image to show the boxes if cropping is enabled
if f == preview_frame
sample_bigImage = bigImage;
end
end
%% --- MEMORY MANAGEMENT ---
if mod(f, 20) == 0 & f < length(subtractedfolder)
disp("Resetting memory at image " + f);
reset(gpuDevice);
clear IMG IMG_Filtered img_filtered ...
M M1 S Tmax D2 T ...
BinaryImage BinaryImage1 BinaryImage2 ...
filtered filteredBinaryImage thresholdValue ...
cc stats cropImg bigImage I_gpu I_fft I_crop M_crop;
end
end % End image loop
%% FINALIZATION
disp("All Images Reconstructed!");
rectime = toc;
if rectime < 60
disp("This took " + num2str(rectime) + " seconds");
elseif rectime < 3600
disp("This took " + num2str(rectime/60.0) + " minutes");
else
disp("This took " + string(floor(rectime/3600)) + " hours and " + string(floor((rectime/3600 - floor(rectime/3600))*60)) + " minutes");
end
%% --- SETTINGS & VISUAL DISPLAY ---
if exist('sample_bigImage', 'var') && exist('sample_BinaryImage', 'var')
figure;
imshow(sample_bigImage, []);
title("Sample Splat with Bounding Boxes (Frame "+ preview_frame + ")");
figure;
imshow(sample_BinaryImage, []);
title("Sample Binary Mask (Frame " + preview_frame + ")");
else
disp('No sample images available for display. Check if crop_flag = 1.');
end
% Settings Display for Refinement
disp('--- Current Settings Used ---');
display(lambda);
display(MaskRadius);
display(ButterworthOrder);
display(thresholdValue1);
display(minParticleSize);
display(paddingNum);
display(filter_size)
