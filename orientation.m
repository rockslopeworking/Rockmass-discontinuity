function [dip,dipdirection]=orientation(normalvector)

% if normalvector(:,3)<0
%         normalvector=-normalvector;
% else
% end
A=normalvector(:,1);
B=normalvector(:,2);
C=normalvector(:,3);

dip=acosd(abs(C));


if A>0
        dipdirection=90-atand(B./A);
else
        dipdirection=270-atand(B./A);
end

end