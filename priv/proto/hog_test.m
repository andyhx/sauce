PositiveSetTest = 'F:/inz/priv/sanity_test/test/pos';
NegativeSetTest = 'F:/inz/priv/sanity_test/test/neg';

PositivesT = dir(PositiveSetTest);

Positives = 0;
FalseNegatives = 0;
for i=3:length(PositivesT)
    ImageFile = [PositiveSetTest, '/', PositivesT(i, 1).name];
    Image = imread(ImageFile);
    ImageGrayscale = rgb2gray(Image);
    
    [Gx, Gy] = hog_gradient(ImageGrayscale);
    [Magnitude, Orientation] = hog_mag_orient(Gx, Gy);
    
    Bins = hog_binning(9, Magnitude, Orientation);
    Bins = Bins / (norm(Bins, 1) + 0.001);
    
    Detected = svmclassify(SVM, Bins(:)');
    Positives = Positives+1;
    if Detected == 0
        FalseNegatives = FalseNegatives+1;
    end
end
fprintf('False negatives: %d\nHit rate: %d/%d = %f%\n', FalseNegatives, Positives-FalseNegatives, Positives,(Positives-FalseNegatives)/(Positives)*100);

NegativesT = dir(NegativeSetTest);
N = length(PositivesT)-2;

Negatives = 0;
FalsePositives = 0;
for i=3:length(NegativesT);
    ImageFile = [NegativeSetTest, '/', NegativesT(i, 1).name];
    Image = imread(ImageFile);
    ImageGrayscale = rgb2gray(Image);
    
    [Gx, Gy] = hog_gradient(ImageGrayscale);
    [Magnitude, Orientation] = hog_mag_orient(Gx, Gy);
    
    Bins = hog_binning(9, Magnitude, Orientation);
    Bins = Bins / (norm(Bins, 1) + 0.001);
    
    Detected = svmclassify(SVM, Bins(:)');
    Negatives = Negatives+1;
    if Detected == 1
        FalsePositives = FalsePositives+1;
    end
end
fprintf('False positives: %d\nHit rate: %d/%d = %f%\n', FalsePositives, Negatives-FalsePositives, Negatives, (Negatives-FalsePositives)/(Negatives)*100);

