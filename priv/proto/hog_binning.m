function [ Bins ] = hog_binning( BinsNo, Mag, Orient )

[H, W] = size(Mag);
XCells = W/8;
YCells = H/8;
Bins = zeros(XCells*YCells, BinsNo);

AngleLut = zeros(BinsNo, 1);
XLut = zeros(XCells, 1);
YLut = zeros(YCells, 1);

for i=1:BinsNo
    AngleLut(i) = pi/BinsNo * (i-1);
end
for i=1:W/8
    XLut(i) = W/XCells * (i-1) + 1;
end
for i=1:H/8
    YLut(i) = H/YCells * (i-1) + 1;
end

for i=1:H
    for j=1:W
        % angle interpolation
        Angle = Orient(i,j);
        Bin = floor(Angle/pi * BinsNo);
        if Bin == 0 || Bin >= BinsNo
            AngleBin2 = BinsNo;
            AngleBin1 = BinsNo-1;
        else
            AngleBin2 = Bin+1;
            AngleBin1 = Bin;
        end  
        Angle2 = AngleLut(AngleBin2);
        Angle1 = AngleLut(AngleBin1);
        Ca1 = (Angle - Angle1)/(Angle2 - Angle1);
        Ca2 = (Angle2 - Angle)/(Angle2 - Angle1);

        
        % histograms interpolation
        X = j;
        XBin = floor((X/W) * XCells);
        if XBin == 0 || XBin >= XCells
            XBin2 = XCells;
            XBin1 = XCells-1;
        else
            XBin2 = XBin+1;
            XBin1 = XBin;
        end
        
        X2 = XLut(XBin2);
        X1 = XLut(XBin1);
        
        Y = i;
        YBin = floor((Y/H) * YCells);
        if YBin == 0 || YBin >= YCells
            YBin2 = YCells;
            YBin1 = YCells-1;
        else
            YBin2 = YBin+1;
            YBin1 = YBin;
        end
        Y2 = YLut(YBin2);
        Y1 = YLut(YBin1);
        
        H1 = (YBin1-1) * XCells + XBin1;
        H2 = (YBin1-1) * XCells + XBin2;
        H3 = YBin1 * XCells + XBin1;
        H4 = YBin1 * XCells + XBin2;
        
        CaX1 = (X-X1)/(X2-X1);
        CaX2 = (X2-X)/(X2-X1);
        CaY1 = (Y-Y1)/(Y2-Y1);
        CaY2 = (Y2-Y)/(Y2-Y1);
        M = Mag(i,j);
        
        Bins(H1, AngleBin1) = Bins(H1, AngleBin1) + CaX1*CaY1*Ca1*M;
        Bins(H1, AngleBin2) = Bins(H1, AngleBin2) + CaX1*CaY1*Ca2*M;
        
        Bins(H2, AngleBin1) = Bins(H2, AngleBin1) + CaX2*CaY1*Ca1*M;
        Bins(H2, AngleBin2) = Bins(H2, AngleBin2) + CaX2*CaY1*Ca2*M;
        
        Bins(H3, AngleBin1) = Bins(H3, AngleBin1) + CaX1*CaY2*Ca1*M;
        Bins(H3, AngleBin2) = Bins(H3, AngleBin2) + CaX1*CaY2*Ca2*M;
        
        Bins(H4, AngleBin1) = Bins(H4, AngleBin1) + CaX2*CaY2*Ca1*M;
        Bins(H4, AngleBin2) = Bins(H4, AngleBin2) + CaX2*CaY2*Ca2*M;
    end
end
end

