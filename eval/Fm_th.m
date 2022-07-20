function [all_f_th, all_th] = Fm_th(sMap, gtMap, gtsize)

sMap = 255 * sMap;
all_f_th = zeros(256,1);
all_th    = zeros(256,1);
for threshold = 0:255 
    Label3 = zeros( gtsize );
    Label3(sMap>=threshold ) = 1;
    NumRec   = length( find( Label3==1 ) );
    LabelAnd = Label3 & gtMap;
    NumAnd   = length( find ( LabelAnd==1 ) );
    num_obj  = sum(sum(gtMap));
    if NumAnd == 0
        PreFtem    = 0;
        RecallFtem = 0;
        f_th  = 0;
    else
        PreFtem    = NumAnd/NumRec;
        RecallFtem = NumAnd/num_obj;
        f_th  = ((1.3*PreFtem*RecallFtem)/(0.3*PreFtem+RecallFtem));
        th = threshold;
        all_f_th(threshold+1,:)       = f_th;
        all_th(threshold+1,:)         = th;
    end
    
end