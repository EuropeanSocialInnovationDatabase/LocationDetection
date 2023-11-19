from location_detect_algorithm.location_detection import LocationDetection

ld = LocationDetection('')

pages = [
    {
        "content": "Our company is based in Barcelona and we have new branches in Leon, Spain",
        "url": "www.website.es"

    },
    {
        "content": "We are happy to present our work in Glasgow",
        "url": "www.website.es"
    },
    {
        "content": "Please, vist us in Barcelona",
        "url": "www.website.es"
    }
]

loc_info = ld.single_project_location_detection(pages)
print(loc_info)
