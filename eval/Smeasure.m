function Q = Smeasure(prediction,GT)

if (~isa(prediction,'double'))
    error('The prediction should be double type...');
end
if ((max(prediction(:))>1) || min(prediction(:))<0)
    error('The prediction should be in the range of [0 1]...');
end
if (~islogical(GT))
    error('GT should be logical type...');
end

y = mean2(GT);

if (y==0)
    x = mean2(prediction);
    Q = 1.0 - x; 
elseif(y==1)
    x = mean2(prediction);
    Q = x; 
else
    alpha = 0.5;
    Q = alpha*S_object(prediction,GT)+(1-alpha)*S_region(prediction,GT);
end

end