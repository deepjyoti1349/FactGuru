class InputHandler:
    def validate_claim(self,claim):
        if not claim or claim.strip()=='':
            return False, "claim cant be empty"
        clean_claim=claim.strip()
        
        if len(clean_claim)<5:
            return False, " claim must be atleast 5 chars long"
        if len(clean_claim)>500:
            return False,"claim must be lesser than 500 chars"
        
        has_letters = any(character.isalpha() for character in clean_claim)
        if not has_letters:
            return False , " claim should contain alphabets"
        return True,""
    def process_input(self,claim):
        is_valid , error = self.validate_claim(claim)
        if not is_valid:
            return "",error
        cleaned=" ".join(claim.split())
        #splitting each word and putting tem together with " " bet them.
        return cleaned,""