function [Q]= wFmeasure(FG,GT)

if (~isa( FG, 'double' ))
    error('FG should be of type: double');
end
if ((max(FG(:))>1) || min(FG(:))<0)
    error('FG should be in the range of [0 1]');
end
if (~islogical(GT))
    error('GT should be of type: logical');
end



dGT = double(GT); 
if max(dGT(:)) == 0
    Q = 0;
    return 
end

E = abs(FG-dGT);


[Dst,IDXT] = bwdist(dGT);

K = fspecial('gaussian',7,5);
Et = E;
Et(~GT)=Et(IDXT(~GT)); 
EA = imfilter(Et,K);
MIN_E_EA = E;
MIN_E_EA(GT & EA<E) = EA(GT & EA<E);

B = ones(size(GT));
B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
Ew = MIN_E_EA.*B;

TPw = sum(dGT(:)) - sum(sum(Ew(GT))); 
FPw = sum(sum(Ew(~GT)));

R = 1- mean2(Ew(GT)); 
P = TPw./(eps+TPw+FPw); 

Q = (2)*(R*P)./(eps+R+P); 

end