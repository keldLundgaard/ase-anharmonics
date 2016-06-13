"""
File which contains a dictionary with all settings for the fitting procedure

The inteded use is to overwrite these default settings as needed in the code
and send the dictionary as an input argument to the fitting constructor.
"""
fit_settings = {
    # Basis fitting function type
    # Implemented basis functions:
    #   legendra_int2basis
    #   trigonometric
    'basistype': 'undefined',

    # Smoothing level.
    # Means smoothing operator will be pdiff times
    #   differentiated from the original baisfunctions.
    # In fitting procedures, the smoothness operator is equal to
    #   identity and basisfunctions adjusted accordingly.
    'pdiff': 2,

    # Number of coefficients to basis for the prior fit
    # Setting this to zero means no prior will be used
    'p_order': 0,

    # Either omega2 is found from one array or with iterative scanning
    #   onesweep
    #   iterative
    'search_method': 'onesweep',

    # Technically only used for periodic basis functions
    # Telling the fitting how many times the 360 degree angles
    #    full period the data is split into
    # 2 => periodic every 180 degrees
    'symnumber': 1,

    # Weighting the derivative impact on fitting. When fitting the curve
    # regular data will always have a comperative weight 1.
    'derivateive_weight': 1,

    # Print 'debug' info to terminal as program runs
    'verbose': 1

}
