function Jc = CalcContactJacobian(model, q, flag_contact, nf)
%CalcContactJacobian - Description
%
% Syntax: Jc = CalcContactJacobian(q, flag_contact)
%
% Long description
    
    k = 0;
    for i=1:model.NC
        if flag_contact(i)~=0
            % Calculate Jacobian 
            J = CalcPointJacobian(model, q, model.idcontact(i), model.contactpoint{i});
            
            % Make Jacobian full rank according to contact model
            if nf==2
                Jci = [J(1, :); J(3, :)]; % only x\z direction
            else 
                if nf==3
                    Jci = J;
                end
            end

            Jc(k*nf+1:k*nf+nf, :) = Jci;
            
            k = k+1;
        end
    end
end