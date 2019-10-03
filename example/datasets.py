domainnet_datasets = {
        'real': '/data/DomainNet/real_train.txt',
        'real-test': '/data/DomainNet/real_test.txt',
        'infograph': '/data/DomainNet/infograph_train.txt',
        'infograph-test': '/data/DomainNet/infograph_test.txt',
        'quickdraw': '/data/DomainNet/quickdraw_train.txt',
        'quickdraw-test': '/data/DomainNet/quickdraw_test.txt',
        'sketch': '/data/DomainNet/sketch_train.txt',
        'sketch-test': '/data/DomainNet/sketch_test.txt',
        }

office_datasets = {
        'amazon': '/data/office/amazon_list.txt',
        'amazon-train': '/data/office/amazon_list_train.txt',
        'amazon-test': '/data/office/amazon_list_test.txt',
        'amazon-20noise': '/data/office/amazon_list_20-noise.txt',
        'amazon-40noise': '/data/office/amazon_list_40-noise.txt',
        'amazon-60noise': '/data/office/amazon_list_60-noise.txt',
        'dslr': '/data/office/dslr_list.txt',
        'dslr-train': '/data/office/dslr_list_train.txt',
        'dslr-test': '/data/office/dslr_list_test.txt',
        'dslr-20noise': '/data/office/dslr_list_20-noise.txt',
        'dslr-40noise': '/data/office/dslr_list_40-noise.txt',
        'dslr-60noise': '/data/office/dslr_list_60-noise.txt',
        'webcam': '/data/office/webcam_list.txt',
        'webcam-train': '/data/office/webcam_list_train.txt',
        'webcam-test': '/data/office/webcam_list_test.txt',
        'webcam-20noise': '/data/office/webcam_list_20-noise.txt',
        'webcam-40noise': '/data/office/webcam_list_40-noise.txt',
        'webcam-60noise': '/data/office/webcam_list_60-noise.txt',
        }

officehome_datasets = {
        'Art': '/data/office-home/Art.txt',
        'Product': '/data/office-home/Product.txt',
        'Clipart': '/data/office-home/Clipart.txt',
        'RealWorld': '/data/office-home/Real_World.txt',
        }

data = {
        'office': office_datasets,
        'officehome': officehome_datasets,
        'domainnet': domainnet_datasets,
        }

num_class = {
        'office': 31,
        'officehome': 65,
        'domainnet': 345,
        }
