function new_sig = distortion(A,y)
yn=y(:,1);
yn=A*yn;
yn(yn>1)=1;
yn(yn<-1)=-1;

new_sig = yn;

if size(y,2)>1
   yn = y(:,2);
   yn = A*yn;
   yn(yn>1) = 1;
   yn(yn<-1) = -1;
   
   new_sig(:,2) = yn;
end