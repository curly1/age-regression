function classNum = determineAgeClass(age)
  
if age <= 14, classNum = 1;
else if age <= 24, classNum = 2;
    else if age <= 54, classNum = 3;
        else if age <= 80, classNum = 4;
            else classNum = 0; warning('classNum not recognized')
            end
        end
    end
end

end