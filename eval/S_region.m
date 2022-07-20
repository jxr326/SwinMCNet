function Q = S_region(prediction,GT)

[X,Y] = centroid(GT);


[GT_1,GT_2,GT_3,GT_4,w1,w2,w3,w4] = divideGT(GT,X,Y);


[prediction_1,prediction_2,prediction_3,prediction_4] = Divideprediction(prediction,X,Y);


Q1 = ssim(prediction_1,GT_1);
Q2 = ssim(prediction_2,GT_2);
Q3 = ssim(prediction_3,GT_3);
Q4 = ssim(prediction_4,GT_4);


Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4;

end


function [X,Y] = centroid(GT)

[rows,cols] = size(GT);

if(sum(GT(:))==0)
    X = round(cols/2);
    Y = round(rows/2);
else
    dGT = double(GT);
    x = ones(rows,1)*(1:cols);
    y = (1:rows)'*ones(1,cols);
    area = sum(dGT(:));
    X = round(sum(sum(dGT.*x))/area);
    Y = round(sum(sum(dGT.*y))/area);
end

end



function [LT,RT,LB,RB,w1,w2,w3,w4] = divideGT(GT,X,Y)


[hei,wid] = size(GT);
area = wid * hei;


LT = GT(1:Y,1:X);
RT = GT(1:Y,X+1:wid);
LB = GT(Y+1:hei,1:X);
RB = GT(Y+1:hei,X+1:wid);


w1 = (X*Y)./area;
w2 = ((wid-X)*Y)./area;
w3 = (X*(hei-Y))./area;
w4 = 1.0 - w1 - w2 - w3;
end


function [LT,RT,LB,RB] = Divideprediction(prediction,X,Y)


[hei,wid] = size(prediction);


LT = prediction(1:Y,1:X);
RT = prediction(1:Y,X+1:wid);
LB = prediction(Y+1:hei,1:X);
RB = prediction(Y+1:hei,X+1:wid);

end

function Q = ssim(prediction,GT)


dGT = double(GT);

[hei,wid] = size(prediction);
N = wid*hei;


x = mean2(prediction);
y = mean2(dGT);


sigma_x2 = sum(sum((prediction - x).^2))./(N - 1 + eps);
sigma_y2 = sum(sum((dGT - y).^2))./(N - 1 + eps);


sigma_xy = sum(sum((prediction - x).*(dGT - y)))./(N - 1 + eps);

alpha = 4 * x * y * sigma_xy;
beta = (x.^2 + y.^2).*(sigma_x2 + sigma_y2);

if(alpha ~= 0)
    Q = alpha./(beta + eps);
elseif(alpha == 0 && beta == 0)
    Q = 1.0;
else
    Q = 0;
end

end