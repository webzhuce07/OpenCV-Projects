function g = shockFilter(f)
% Shock filter the image
% PDE : df/dt = -sgn(del2(f)) x norm(grad(f))
% grad(f) = [fx, fy]^T
% norm(grad(f)) = sqrt(fx.^2 + fy.^2)
% del2(f) = f_x^2 f_xx + 2 f_x f_y f_xy + f_y^2 f_yy (Jia's paper)
% or Laplacian(in general)
% del2() func in matlab calculates laplacian: (fxx + fyy) / 4
% 
% Jiaya Jia: Two-Phase Kernel Estimation for Robust Motion Deblurring
% Vijay Jan 02, 2013

% clc;
% close all;
% 
% load mandrill; % X contains double image
% f = X;

f = double(f);
numIter = 10;
dt = 0.5;
filterMaskSize = 9;

% initialize with smoothed image
filt = fspecial('gaussian',filterMaskSize);
g = conv2(f,filt,'same');

for iter = 1:numIter
    g = conv2(g,filt,'same');
    [gx gy] = gradient(g);
    [gxx gxy] = gradient(gx);
    [gyx gyy] = gradient(gy);
    
    dg_by_dt = -sign(del2(g)) .* sqrt(gx.^2 + gy.^2); 
    
    % del2 uses laplacian; the following is used by Jiaya Jia in his 
    % Two phase kernel estimation paper
    % gx.^2 .* gxx + 2 * gx .* gy .* gxy + gy.^2 .* gyy
    del2g = gx.^2 .* gxx + 2 * gx .* gy .* gxy + gy.^2 .* gyy;
    
    dg_by_dt = -sign(del2g) .* sqrt(gx.^2 + gy.^2); 
    g = g + (dg_by_dt) * dt;
end

 figure, imshow(uint8(f));title('ShockFliterBefore');
 figure, imshow(uint8(g));title('ShockFliterAfter');
end

