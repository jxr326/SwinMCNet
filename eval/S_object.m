function Q = S_object(prediction,GT)

prediction_fg = prediction;
prediction_fg(~GT)=0;
O_FG = Object(prediction_fg,GT);


prediction_bg = 1.0 - prediction;
prediction_bg(GT) = 0;
O_BG = Object(prediction_bg,~GT);


u = mean2(GT);
Q = u * O_FG + (1 - u) * O_BG;

end

function score = Object(prediction,GT)


if isempty(prediction)
    score = 0;
    return;
end
if isinteger(prediction)
    prediction = double(prediction);
end
if (~isa( prediction, 'double' ))
    error('prediction should be of type: double');
end
if ((max(prediction(:))>1) || min(prediction(:))<0)
    error('prediction should be in the range of [0 1]');
end
if(~islogical(GT))
    error('GT should be of type: logical');
end


x = mean2(prediction(GT));


sigma_x = std(prediction(GT));

score = 2.0 * x./(x^2 + 1.0 + sigma_x + eps);
end