function [score]= Emeasure(FM,GT)

FM = mat2gray(FM);
thd = 2 * mean(FM(:));
FM = FM > thd;


FM = logical(FM);
GT = logical(GT);


dFM = double(FM);
dGT = double(GT);


if (sum(dGT(:))==0)
    enhanced_matrix = 1.0 - dFM; 
elseif(sum(~dGT(:))==0)
    enhanced_matrix = dFM; 
else
    
    
    
    align_matrix = AlignmentTerm(dFM,dGT);
    
    enhanced_matrix = EnhancedAlignmentTerm(align_matrix);
end


[w,h] = size(GT);
score = sum(enhanced_matrix(:))./(w*h - 1 + eps);
end


function [align_Matrix] = AlignmentTerm(dFM,dGT)


mu_FM = mean2(dFM);
mu_GT = mean2(dGT);


align_FM = dFM - mu_FM;
align_GT = dGT - mu_GT;


align_Matrix = 2.*(align_GT.*align_FM)./(align_GT.*align_GT + align_FM.*align_FM + eps);

end


function enhanced = EnhancedAlignmentTerm(align_Matrix)
enhanced = ((align_Matrix + 1).^2)/4;
end

