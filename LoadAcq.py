
""" ================================================================
                            ACQ TO MAT
    ================================================================"""

def LoadFiles(filesDir,outputDir):
    files = glob(filesDir)
    for i, file in enumerate(files):
        file_name = os.path.splitext(os.path.basename(file))[0]
        save_path = os.path.join(outputDir, f'{file_name}.mat')

        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        if not os.path.exists(save_path):
            acq = nk.read_acqknowledge(file, sampling_rate='max')
            signals = acq[0] # Assuming acq[0] contains the signals
            signals = signals.to_numpy()
            Fs = acq[1]
            data_to_save = {'Sig': signals, 'Fs': Fs}
            sio.savemat(save_path, data_to_save, long_field_names=True, do_compression=True)

