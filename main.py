import argparse

from location_detect_algorithm.location_detection import LocationDetection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--collection', default="all_newcrawl_combined_271022", help="MongoDB Collection name")
    parser.add_argument('-p', '--projects', default=None,
                        help="Path to a text file of project IDs to predict their locations")
    args = parser.parse_args()
    loc_detect = LocationDetection(args.collection)
    loc_detect.predict_all_projects_loc(args.projects)
